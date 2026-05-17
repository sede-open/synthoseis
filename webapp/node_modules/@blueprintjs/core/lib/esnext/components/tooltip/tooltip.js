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
export class Tooltip extends AbstractPureComponent {
    static displayName = `${DISPLAYNAME_PREFIX}.Tooltip`;
    static defaultProps = {
        compact: false,
        hoverCloseDelay: 0,
        hoverOpenDelay: 100,
        interactionKind: "hover-target",
        minimal: false,
        transitionDuration: 100,
    };
    popoverRef = React.createRef();
    render() {
        // if we have an ancestor TooltipContext, we should take its state into account in this render path,
        // it was likely created by a parent ContextMenu
        return (React.createElement(TooltipContext.Consumer, null, ([state]) => React.createElement(TooltipProvider, { ...state }, this.renderPopover)));
    }
    reposition() {
        this.popoverRef.current?.reposition();
    }
    // any descendant ContextMenus may update this ctxState
    renderPopover = (ctxState) => {
        const { children, compact, disabled, intent, popoverClassName, ...restProps } = this.props;
        const popoverClasses = classNames(Classes.TOOLTIP, Classes.intentClass(intent), popoverClassName, {
            [Classes.COMPACT]: compact,
        });
        return (React.createElement(Popover, { modifiers: {
                arrow: {
                    enabled: !this.props.minimal,
                },
                offset: {
                    options: {
                        offset: [0, TOOLTIP_ARROW_SVG_SIZE / 2],
                    },
                },
            }, ...restProps, autoFocus: false, canEscapeKeyClose: false, disabled: ctxState.forceDisabled ?? disabled, enforceFocus: false, lazy: true, popoverClassName: popoverClasses, portalContainer: this.props.portalContainer, ref: this.popoverRef }, children));
    };
}
//# sourceMappingURL=tooltip.js.map