"use strict";
/*
 * Copyright 2021 Palantir Technologies, Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
Object.defineProperty(exports, "__esModule", { value: true });
exports.Tooltip = void 0;
var tslib_1 = require("tslib");
var classnames_1 = tslib_1.__importDefault(require("classnames"));
var React = tslib_1.__importStar(require("react"));
var common_1 = require("../../common");
var Classes = tslib_1.__importStar(require("../../common/classes"));
// eslint-disable-next-line import/no-cycle
var popover_1 = require("../popover/popover");
var popoverArrow_1 = require("../popover/popoverArrow");
var tooltipContext_1 = require("../popover/tooltipContext");
/**
 * Tooltip component.
 *
 * @see https://blueprintjs.com/docs/#core/components/tooltip
 */
var Tooltip = /** @class */ (function (_super) {
    tslib_1.__extends(Tooltip, _super);
    function Tooltip() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.popoverRef = React.createRef();
        // any descendant ContextMenus may update this ctxState
        _this.renderPopover = function (ctxState) {
            var _a;
            var _b;
            var _c = _this.props, children = _c.children, compact = _c.compact, disabled = _c.disabled, intent = _c.intent, popoverClassName = _c.popoverClassName, restProps = tslib_1.__rest(_c, ["children", "compact", "disabled", "intent", "popoverClassName"]);
            var popoverClasses = (0, classnames_1.default)(Classes.TOOLTIP, Classes.intentClass(intent), popoverClassName, (_a = {},
                _a[Classes.COMPACT] = compact,
                _a));
            return (React.createElement(popover_1.Popover, tslib_1.__assign({ modifiers: {
                    arrow: {
                        enabled: !_this.props.minimal,
                    },
                    offset: {
                        options: {
                            offset: [0, popoverArrow_1.TOOLTIP_ARROW_SVG_SIZE / 2],
                        },
                    },
                } }, restProps, { autoFocus: false, canEscapeKeyClose: false, disabled: (_b = ctxState.forceDisabled) !== null && _b !== void 0 ? _b : disabled, enforceFocus: false, lazy: true, popoverClassName: popoverClasses, portalContainer: _this.props.portalContainer, ref: _this.popoverRef }), children));
        };
        return _this;
    }
    Tooltip.prototype.render = function () {
        var _this = this;
        // if we have an ancestor TooltipContext, we should take its state into account in this render path,
        // it was likely created by a parent ContextMenu
        return (React.createElement(tooltipContext_1.TooltipContext.Consumer, null, function (_a) {
            var state = _a[0];
            return React.createElement(tooltipContext_1.TooltipProvider, tslib_1.__assign({}, state), _this.renderPopover);
        }));
    };
    Tooltip.prototype.reposition = function () {
        var _a;
        (_a = this.popoverRef.current) === null || _a === void 0 ? void 0 : _a.reposition();
    };
    Tooltip.displayName = "".concat(common_1.DISPLAYNAME_PREFIX, ".Tooltip");
    Tooltip.defaultProps = {
        compact: false,
        hoverCloseDelay: 0,
        hoverOpenDelay: 100,
        interactionKind: "hover-target",
        minimal: false,
        transitionDuration: 100,
    };
    return Tooltip;
}(common_1.AbstractPureComponent));
exports.Tooltip = Tooltip;
//# sourceMappingURL=tooltip.js.map