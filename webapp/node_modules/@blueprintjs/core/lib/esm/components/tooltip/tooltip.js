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
import { __assign, __extends, __rest } from "tslib";
import classNames from "classnames";
import * as React from "react";
import { AbstractPureComponent, DISPLAYNAME_PREFIX } from "../../common";
import * as Classes from "../../common/classes";
// eslint-disable-next-line import/no-cycle
import { Popover } from "../popover/popover";
import { TOOLTIP_ARROW_SVG_SIZE } from "../popover/popoverArrow";
import { TooltipContext, TooltipProvider } from "../popover/tooltipContext";
/**
 * Tooltip component.
 *
 * @see https://blueprintjs.com/docs/#core/components/tooltip
 */
var Tooltip = /** @class */ (function (_super) {
    __extends(Tooltip, _super);
    function Tooltip() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.popoverRef = React.createRef();
        // any descendant ContextMenus may update this ctxState
        _this.renderPopover = function (ctxState) {
            var _a;
            var _b;
            var _c = _this.props, children = _c.children, compact = _c.compact, disabled = _c.disabled, intent = _c.intent, popoverClassName = _c.popoverClassName, restProps = __rest(_c, ["children", "compact", "disabled", "intent", "popoverClassName"]);
            var popoverClasses = classNames(Classes.TOOLTIP, Classes.intentClass(intent), popoverClassName, (_a = {},
                _a[Classes.COMPACT] = compact,
                _a));
            return (React.createElement(Popover, __assign({ modifiers: {
                    arrow: {
                        enabled: !_this.props.minimal,
                    },
                    offset: {
                        options: {
                            offset: [0, TOOLTIP_ARROW_SVG_SIZE / 2],
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
        return (React.createElement(TooltipContext.Consumer, null, function (_a) {
            var state = _a[0];
            return React.createElement(TooltipProvider, __assign({}, state), _this.renderPopover);
        }));
    };
    Tooltip.prototype.reposition = function () {
        var _a;
        (_a = this.popoverRef.current) === null || _a === void 0 ? void 0 : _a.reposition();
    };
    Tooltip.displayName = "".concat(DISPLAYNAME_PREFIX, ".Tooltip");
    Tooltip.defaultProps = {
        compact: false,
        hoverCloseDelay: 0,
        hoverOpenDelay: 100,
        interactionKind: "hover-target",
        minimal: false,
        transitionDuration: 100,
    };
    return Tooltip;
}(AbstractPureComponent));
export { Tooltip };
//# sourceMappingURL=tooltip.js.map