/**
 * Bypass Groups
 */
import { app } from "../../scripts/app.js";

// ---- Config: rename for your pack ------------------------------------------------------------
const NODE_TYPE = "Bypass Groups";
const NODE_TITLE = "Bypass Groups";
const NODE_CATEGORY = "custom";
const EXTENSION_NAME = "codinghero.BypassGroups";

const MODE_ON = LiteGraph.ALWAYS; // 0
const MODE_OFF = 4; // ComfyUI's "bypass"

const PROP_SORT = "sort";
const PROP_SORT_ALPHA = "customSortAlphabet";
const PROP_MATCH_COLORS = "matchColors";
const PROP_MATCH_TITLE = "matchTitle";
const PROP_SHOW_NAV = "showNav";
const PROP_SHOW_ALL_GRAPHS = "showAllGraphs";
const PROP_RESTRICTION = "toggleRestriction"; // "default" | "max one" | "always one"

// ---- Utils -----------------------------------------------------------------------------------

/** Depth-first walk over nodes, descending into subgraph nodes. */
function walkNodesDepthFirst(nodeOrNodes, fn) {
  const stack = (Array.isArray(nodeOrNodes) ? [...nodeOrNodes] : [nodeOrNodes]).reverse();
  while (stack.length) {
    const node = stack.pop();
    fn(node);
    if (node.isSubgraphNode?.() && node.subgraph) {
      const kids = node.subgraph.nodes;
      for (let i = kids.length - 1; i >= 0; i--) stack.push(kids[i]);
    }
  }
}

function changeModeOfNodes(nodes, mode) {
  walkNodesDepthFirst(nodes, (n) => {
    n.mode = mode;
  });
}

/** group._nodes can contain stale entries; _children is the reliable source. */
function getGroupNodes(group) {
  return Array.from(group._children).filter((c) => c instanceof LGraphNode);
}

/** Same node id can exist in different subgraphs, so key on graph id too. */
function graphDependantNodeKey(node) {
  const graph = node.graph ?? app.graph;
  return `${graph.id}:${node.id}`;
}

function isLowQuality() {
  return (app.canvas.ds?.scale || 1) <= 0.5;
}

function fitString(ctx, str, maxWidth) {
  const ellipsis = "…";
  const width = ctx.measureText(str).width;
  const ellipsisWidth = ctx.measureText(ellipsis).width;
  if (width <= maxWidth || width <= ellipsisWidth) return str;
  let lo = 0;
  let hi = str.length;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (ctx.measureText(str.substring(0, mid)).width <= maxWidth - ellipsisWidth) lo = mid;
    else hi = mid - 1;
  }
  return str.substring(0, lo) + ellipsis;
}

/** Draws the pill background for a custom widget row and returns layout data. */
function drawNodeWidget(ctx, { width, height, posY }) {
  const lowQuality = isLowQuality();
  const data = {
    width,
    height,
    posY,
    lowQuality,
    margin: 15,
    colorOutline: LiteGraph.WIDGET_OUTLINE_COLOR,
    colorBackground: LiteGraph.WIDGET_BGCOLOR,
    colorText: LiteGraph.WIDGET_TEXT_COLOR,
    colorTextSecondary: LiteGraph.WIDGET_SECONDARY_TEXT_COLOR,
  };
  ctx.strokeStyle = data.colorOutline;
  ctx.fillStyle = data.colorBackground;
  ctx.beginPath();
  ctx.roundRect(
    data.margin,
    data.posY,
    data.width - data.margin * 2,
    data.height,
    lowQuality ? [0] : [height * 0.5],
  );
  ctx.fill();
  if (!lowQuality) ctx.stroke();
  return data;
}

// ---- Service: shared group discovery / refresh loop ------------------------------------------

class BypassGroupsService {
  msThreshold = 400;
  msLastUnsorted = 0;
  msLastAlpha = 0;
  msLastPosition = 0;

  groupsUnsorted = [];
  groupsSortedAlpha = [];
  groupsSortedPosition = [];

  nodes = [];

  runScheduledForMs = null;
  runScheduleTimeout = null;
  runScheduleAnimation = null;

  cachedNodeBoundings = null;

  addNode(node) {
    this.nodes.push(node);
    // Deferred: on add, the node may not have its cloned properties yet.
    this.scheduleRun(8);
  }

  removeNode(node) {
    const i = this.nodes.indexOf(node);
    if (i > -1) this.nodes.splice(i, 1);
    // No more nodes -> probably a canvas clear; drop cached state.
    if (!this.nodes.length) {
      this.clearScheduledRun();
      this.groupsUnsorted = [];
      this.groupsSortedAlpha = [];
      this.groupsSortedPosition = [];
    }
  }

  run() {
    if (!this.runScheduledForMs) return;
    for (const node of this.nodes) node.refreshWidgets();
    this.clearScheduledRun();
    this.scheduleRun();
  }

  scheduleRun(ms = 500) {
    // A shorter request preempts a longer pending one.
    if (this.runScheduledForMs && ms < this.runScheduledForMs) this.clearScheduledRun();
    if (!this.runScheduledForMs && this.nodes.length) {
      this.runScheduledForMs = ms;
      this.runScheduleTimeout = setTimeout(() => {
        this.runScheduleAnimation = requestAnimationFrame(() => this.run());
      }, ms);
    }
  }

  clearScheduledRun() {
    this.runScheduleTimeout && clearTimeout(this.runScheduleTimeout);
    this.runScheduleAnimation && cancelAnimationFrame(this.runScheduleAnimation);
    this.runScheduleTimeout = null;
    this.runScheduleAnimation = null;
    this.runScheduledForMs = null;
  }

  /** Bounds for every node, cached ~50ms so N groups don't each re-walk the graph. */
  getBoundingsForAllNodes() {
    if (!this.cachedNodeBoundings) {
      const acc = {};
      walkNodesDepthFirst(app.graph._nodes, (node) => {
        let bounds = node.getBounding();
        // Zeroed bounds = subgraph node that hasn't rendered yet.
        if (bounds[0] === 0 && bounds[1] === 0 && bounds[2] === 0 && bounds[3] === 0) {
          const ctx = node.graph?.primaryCanvas?.canvas.getContext("2d");
          if (ctx) {
            node.updateArea(ctx);
            bounds = node.getBounding();
          }
        }
        acc[graphDependantNodeKey(node)] = bounds;
      });
      this.cachedNodeBoundings = acc;
      setTimeout(() => {
        this.cachedNodeBoundings = null;
      }, 50);
    }
    return this.cachedNodeBoundings;
  }

  /**
   * Replacement for group.recomputeInsideNodes(). LiteGraph's version iterates all nodes per
   * group; this shares one bounds pass across all groups. Membership = node center inside group.
   */
  recomputeInsideNodesForGroup(group) {
    if (app.canvas.isDragging) return;
    const bounds = this.getBoundingsForAllNodes();
    group._children.clear();
    group.nodes.length = 0;
    const gb = group._bounding;
    for (const node of group.graph.nodes) {
      const nb = bounds[graphDependantNodeKey(node)];
      if (!nb) continue;
      const cx = nb[0] + nb[2] * 0.5;
      const cy = nb[1] + nb[3] * 0.5;
      if (cx >= gb[0] && cx < gb[0] + gb[2] && cy >= gb[1] && cy < gb[1] + gb[3]) {
        group._children.add(node);
        group.nodes.push(node);
      }
    }
  }

  getGroupsUnsorted(now) {
    const canvas = app.canvas;
    const graph = canvas.getCurrentGraph() ?? app.graph;
    if (
      !canvas.selected_group_moving &&
      (!this.groupsUnsorted.length || now - this.msLastUnsorted > this.msThreshold)
    ) {
      this.groupsUnsorted = [...graph._groups];
      const subgraphs = graph.subgraphs?.values();
      if (subgraphs) {
        let s;
        while ((s = subgraphs.next().value)) this.groupsUnsorted.push(...(s.groups ?? []));
      }
      for (const group of this.groupsUnsorted) {
        this.recomputeInsideNodesForGroup(group);
        group.fgb_hasAnyActiveNode = getGroupNodes(group).some((n) => n.mode === MODE_ON);
      }
      this.msLastUnsorted = now;
    }
    return this.groupsUnsorted;
  }

  getGroupsAlpha(now) {
    if (!this.groupsSortedAlpha.length || now - this.msLastAlpha > this.msThreshold) {
      this.groupsSortedAlpha = [...this.getGroupsUnsorted(now)].sort((a, b) =>
        a.title.localeCompare(b.title),
      );
      this.msLastAlpha = now;
    }
    return this.groupsSortedAlpha;
  }

  getGroupsPosition(now) {
    if (!this.groupsSortedPosition.length || now - this.msLastPosition > this.msThreshold) {
      this.groupsSortedPosition = [...this.getGroupsUnsorted(now)].sort((a, b) => {
        // y then x, bucketed to 30px so near-aligned groups sort left-to-right.
        const aY = Math.floor(a._pos[1] / 30);
        const bY = Math.floor(b._pos[1] / 30);
        if (aY === bY) return Math.floor(a._pos[0] / 30) - Math.floor(b._pos[0] / 30);
        return aY - bY;
      });
      this.msLastPosition = now;
    }
    return this.groupsSortedPosition;
  }

  getGroups(sort) {
    const now = +new Date();
    if (sort === "alphanumeric") return this.getGroupsAlpha(now);
    if (sort === "position") return this.getGroupsPosition(now);
    return this.getGroupsUnsorted(now);
  }
}

const SERVICE = new BypassGroupsService();

// ---- Toggle row widget -----------------------------------------------------------------------

class BypassGroupsToggleRowWidget {
  type = "custom";
  name = "FGB_TOGGLE_AND_NAV";
  value = { toggled: false }; // Object, not array: arrays get treated as links on serialize.
  options = { on: "yes", off: "no" };
  label = "";
  y = 0;
  last_y = 0;
  disabled = false;

  constructor(group, node) {
    this.group = group;
    this.node = node;
  }

  get toggled() {
    return this.value.toggled;
  }
  set toggled(v) {
    this.value.toggled = v;
  }

  doModeChange(force, skipOtherNodeCheck) {
    this.group.recomputeInsideNodes();
    const hasAnyActive = getGroupNodes(this.group).some((n) => n.mode === MODE_ON);
    let newValue = force != null ? force : !hasAnyActive;
    if (skipOtherNodeCheck !== true) {
      const restriction = this.node.properties?.[PROP_RESTRICTION];
      if (newValue && restriction?.includes(" one")) {
        for (const w of this.node.widgets) {
          if (w instanceof BypassGroupsToggleRowWidget) w.doModeChange(false, true);
        }
      } else if (!newValue && restriction === "always one") {
        newValue = this.node.widgets.every((w) => !w.value || w === this);
      }
    }
    changeModeOfNodes(getGroupNodes(this.group), newValue ? MODE_ON : MODE_OFF);
    this.group.fgb_hasAnyActiveNode = newValue;
    this.toggled = newValue;
    this.group.graph?.setDirtyCanvas(true, false);
  }

  toggle(value) {
    value = value == null ? !this.toggled : value;
    if (value !== this.toggled) {
      this.value.toggled = value;
      this.doModeChange();
    }
  }

  draw(ctx, node, width, posY, height) {
    const wd = drawNodeWidget(ctx, { width, height, posY });
    const showNav = node.properties?.[PROP_SHOW_NAV] !== false;

    // Render right-to-left; label gets whatever space remains.
    let x = wd.width - wd.margin;

    if (!wd.lowQuality && showNav) {
      x -= 7;
      const midY = wd.posY + wd.height * 0.5;
      ctx.fillStyle = ctx.strokeStyle = "#89A";
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      const arrow = new Path2D(`M${x} ${midY} l -7 6 v -3 h -7 v -6 h 7 v -3 z`);
      ctx.fill(arrow);
      ctx.stroke(arrow);
      x -= 14;

      x -= 7;
      ctx.strokeStyle = wd.colorOutline;
      ctx.stroke(new Path2D(`M ${x} ${wd.posY} v ${wd.height}`));
    } else if (wd.lowQuality && showNav) {
      x -= 28;
    }

    // Toggle dot
    x -= 7;
    ctx.fillStyle = this.toggled ? "#89A" : "#333";
    ctx.beginPath();
    const r = height * 0.36;
    ctx.arc(x - r, posY + height * 0.5, r, 0, Math.PI * 2);
    ctx.fill();
    x -= r * 2;

    if (!wd.lowQuality) {
      x -= 4;
      ctx.textAlign = "right";
      ctx.fillStyle = this.toggled ? wd.colorText : wd.colorTextSecondary;
      const on = this.options.on || "true";
      const off = this.options.off || "false";
      ctx.fillText(this.toggled ? on : off, x, posY + height * 0.7);
      x -= Math.max(ctx.measureText(on).width, ctx.measureText(off).width);

      x -= 7;
      ctx.textAlign = "left";
      const maxLabelWidth = wd.width - wd.margin - 10 - (wd.width - x);
      if (this.label != null) {
        ctx.fillText(fitString(ctx, this.label, maxLabelWidth), wd.margin + 10, posY + height * 0.7);
      }
    }
  }

  serializeValue() {
    return this.value;
  }

  mouse(event, pos, node) {
    if (event.type === "pointerdown") {
      if (node.properties?.[PROP_SHOW_NAV] !== false && pos[0] >= node.size[0] - 15 - 28 - 1) {
        const canvas = app.canvas;
        if (!isLowQuality()) {
          // Nav arrow: center on the group and zoom to fit (never zoom in past current).
          canvas.centerOnNode(this.group);
          const zoomCurrent = canvas.ds?.scale || 1;
          const zoomX = canvas.canvas.width / this.group._size[0] - 0.02;
          const zoomY = canvas.canvas.height / this.group._size[1] - 0.02;
          canvas.setZoom(Math.min(zoomCurrent, zoomX, zoomY), [
            canvas.canvas.width / 2,
            canvas.canvas.height / 2,
          ]);
          canvas.setDirty(true, true);
        }
      } else {
        this.toggle();
      }
    }
    return true;
  }
}

// ---- Node ------------------------------------------------------------------------------------

class BypassGroups extends LGraphNode {
  static type = NODE_TYPE;
  static title = NODE_TITLE;
  static category = NODE_CATEGORY;
  static exposedActions = ["Bypass all", "Enable all", "Toggle all"];

  static "@matchColors" = { type: "string" };
  static "@matchTitle" = { type: "string" };
  static "@showNav" = { type: "boolean" };
  static "@showAllGraphs" = { type: "boolean" };
  static "@sort" = { type: "combo", values: ["position", "alphanumeric", "custom alphabet"] };
  static "@customSortAlphabet" = { type: "string" };
  static "@toggleRestriction" = { type: "combo", values: ["default", "max one", "always one"] };

  comfyClass = NODE_TYPE;
  isVirtualNode = true; // Not sent to the backend / not part of the prompt.
  serialize_widgets = false; // Rows are rebuilt from the groups on load.
  debouncerTempWidth = 0;
  tempSize = null;

  constructor(title = BypassGroups.title) {
    super(title);
    this.widgets = this.widgets || [];
    this.properties = this.properties || {};
    this.properties[PROP_MATCH_COLORS] = "";
    this.properties[PROP_MATCH_TITLE] = "";
    this.properties[PROP_SHOW_NAV] = true;
    this.properties[PROP_SHOW_ALL_GRAPHS] = true;
    this.properties[PROP_SORT] = "position";
    this.properties[PROP_SORT_ALPHA] = "";
    this.properties[PROP_RESTRICTION] = "default";
  }

  // LiteGraph doesn't deep-clone properties.
  clone() {
    const cloned = super.clone();
    if (cloned?.properties && window.structuredClone) {
      cloned.properties = structuredClone(cloned.properties);
    }
    cloned.graph = this.graph;
    return cloned;
  }

  onAdded() {
    SERVICE.addNode(this);
  }

  onRemoved() {
    SERVICE.removeNode(this);
  }

  refreshWidgets() {
    let sort = this.properties?.[PROP_SORT] || "position";
    let customAlphabet = null;
    if (sort === "custom alphabet") {
      const str = this.properties?.[PROP_SORT_ALPHA]?.replace(/\n/g, "");
      if (str && str.trim()) {
        customAlphabet = str.includes(",")
          ? str.toLocaleLowerCase().split(",")
          : str.toLocaleLowerCase().trim().split("");
      }
      if (!customAlphabet?.length) {
        sort = "alphanumeric";
        customAlphabet = null;
      }
    }

    const groups = [...SERVICE.getGroups(sort)];
    // Service pre-sorts alphanumeric/position; custom alphabet is per-node.
    if (customAlphabet?.length) {
      groups.sort((a, b) => {
        let aIdx = -1;
        let bIdx = -1;
        for (const [i, alpha] of customAlphabet.entries()) {
          if (aIdx < 0 && a.title.toLocaleLowerCase().startsWith(alpha)) aIdx = i;
          if (bIdx < 0 && b.title.toLocaleLowerCase().startsWith(alpha)) bIdx = i;
          if (aIdx > -1 && bIdx > -1) break;
        }
        if (aIdx > -1 && bIdx > -1) {
          return aIdx === bIdx ? a.title.localeCompare(b.title) : aIdx - bIdx;
        }
        if (aIdx > -1) return -1;
        if (bIdx > -1) return 1;
        return a.title.localeCompare(b.title);
      });
    }

    // Color filter: accepts Comfy color names (red, pale_blue) or hex (#abc / #aabbcc).
    const normHex = (c) => {
      c = c.replace("#", "").trim().toLocaleLowerCase();
      if (c.length === 3) c = c.replace(/(.)(.)(.)/, "$1$1$2$2$3$3");
      return `#${c}`;
    };
    const filterColors = (this.properties?.[PROP_MATCH_COLORS]?.split(",") || [])
      .filter((c) => c.trim())
      .map((c) => {
        c = c.trim().toLocaleLowerCase();
        if (LGraphCanvas.node_colors[c]) c = LGraphCanvas.node_colors[c].groupcolor;
        return normHex(c);
      });

    let index = 0;
    for (const group of groups) {
      if (filterColors.length) {
        if (!group.color) continue;
        if (!filterColors.includes(normHex(group.color))) continue;
      }
      if (this.properties?.[PROP_MATCH_TITLE]?.trim()) {
        try {
          if (!new RegExp(this.properties[PROP_MATCH_TITLE], "i").exec(group.title)) continue;
        } catch (e) {
          console.error(e);
          continue;
        }
      }
      if (!this.properties?.[PROP_SHOW_ALL_GRAPHS] && group.graph !== app.canvas.getCurrentGraph()) {
        continue;
      }

      let isDirty = false;
      const label = `Enable ${group.title}`;
      let widget = this.widgets.find((w) => w.label === label);
      if (!widget) {
        // LiteGraph mangles size when a widget is added; stash it for computeSize.
        this.tempSize = [...this.size];
        widget = this.addCustomWidget(new BypassGroupsToggleRowWidget(group, this));
        this.setSize(this.computeSize());
        isDirty = true;
      }
      if (widget.label !== label) {
        widget.label = label;
        isDirty = true;
      }
      if (group.fgb_hasAnyActiveNode != null && widget.toggled !== group.fgb_hasAnyActiveNode) {
        widget.toggled = group.fgb_hasAnyActiveNode;
        isDirty = true;
      }
      if (this.widgets[index] !== widget) {
        const oldIndex = this.widgets.indexOf(widget);
        this.widgets.splice(index, 0, this.widgets.splice(oldIndex, 1)[0]);
        isDirty = true;
      }
      if (isDirty) this.setDirtyCanvas(true, false);
      index++;
    }

    // Anything past `index` is a stale row (group deleted/renamed/filtered out).
    while (this.widgets.length > index) {
      this.removeWidget(this.widgets[this.widgets.length - 1]);
    }
  }

  computeSize(out) {
    const size = super.computeSize(out);
    if (this.tempSize) {
      size[0] = Math.max(this.tempSize[0], size[0]);
      size[1] = Math.max(this.tempSize[1], size[1]);
      // computeSize gets hit repeatedly; debounce the clear.
      this.debouncerTempWidth && clearTimeout(this.debouncerTempWidth);
      this.debouncerTempWidth = setTimeout(() => {
        this.tempSize = null;
      }, 32);
    }
    setTimeout(() => this.graph?.setDirtyCanvas(true, true), 16);
    return size;
  }

  async handleAction(action) {
    const restriction = this.properties?.[PROP_RESTRICTION];
    if (action === "Bypass all") {
      const alwaysOne = restriction === "always one";
      for (const [i, w] of this.widgets.entries()) w.doModeChange(alwaysOne && !i, true);
    } else if (action === "Enable all") {
      const onlyOne = restriction.includes(" one");
      for (const [i, w] of this.widgets.entries()) w.doModeChange(!(onlyOne && i > 0), true);
    } else if (action === "Toggle all") {
      const onlyOne = restriction.includes(" one");
      let foundOne = false;
      for (const w of this.widgets) {
        const newValue = onlyOne && foundOne ? false : !w.value;
        foundOne = foundOne || newValue;
        w.doModeChange(newValue, true);
      }
      if (!foundOne && restriction === "always one") {
        this.widgets[this.widgets.length - 1]?.doModeChange(true, true);
      }
    }
  }
}

app.registerExtension({
  name: EXTENSION_NAME,
  registerCustomNodes() {
    LiteGraph.registerNodeType(BypassGroups.type, BypassGroups);
    // Comfy resets `category` on registration; set it after.
    BypassGroups.category = NODE_CATEGORY;
  },
  loadedGraphNode(node) {
    if (node.type === BypassGroups.type) node.tempSize = [...node.size];
  },
});
