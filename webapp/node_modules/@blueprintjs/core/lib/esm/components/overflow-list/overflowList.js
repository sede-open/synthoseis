/*
 * Copyright 2018 Palantir Technologies, Inc. All rights reserved.
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
import { __extends } from "tslib";
import classNames from "classnames";
import * as React from "react";
import { Boundary, Classes, DISPLAYNAME_PREFIX } from "../../common";
import { OVERFLOW_LIST_OBSERVE_PARENTS_CHANGED } from "../../common/errors";
import { shallowCompareKeys } from "../../common/utils";
import { ResizeSensor } from "../resize-sensor/resizeSensor";
/**
 * Overflow list component.
 *
 * @see https://blueprintjs.com/docs/#core/components/overflow-list
 */
var OverflowList = /** @class */ (function (_super) {
    __extends(OverflowList, _super);
    function OverflowList() {
        var _this = _super !== null && _super.apply(this, arguments) || this;
        _this.state = {
            chopSize: _this.defaultChopSize(),
            lastChopSize: null,
            lastOverflowCount: 0,
            overflow: [],
            repartitioning: false,
            visible: _this.props.items,
        };
        _this.spacer = null;
        _this.resize = function () {
            _this.repartition();
        };
        return _this;
    }
    OverflowList.ofType = function () {
        return OverflowList;
    };
    OverflowList.prototype.componentDidMount = function () {
        this.repartition();
    };
    OverflowList.prototype.shouldComponentUpdate = function (nextProps, nextState) {
        // We want this component to always re-render, even when props haven't changed, so that
        // changes in the renderers' behavior can be reflected.
        // The following statement prevents re-rendering only in the case where the state changes
        // identity (i.e. setState was called), but the state is still the same when
        // shallow-compared to the previous state. Original context: https://github.com/palantir/blueprint/pull/3278.
        // We also ensure that we re-render if the props DO change (which isn't necessarily accounted for by other logic).
        return this.props !== nextProps || !(this.state !== nextState && shallowCompareKeys(this.state, nextState));
    };
    OverflowList.prototype.componentDidUpdate = function (prevProps, prevState) {
        var _a, _b;
        if (prevProps.observeParents !== this.props.observeParents) {
            console.warn(OVERFLOW_LIST_OBSERVE_PARENTS_CHANGED);
        }
        if (prevProps.collapseFrom !== this.props.collapseFrom ||
            prevProps.items !== this.props.items ||
            prevProps.minVisibleItems !== this.props.minVisibleItems ||
            prevProps.overflowRenderer !== this.props.overflowRenderer ||
            prevProps.alwaysRenderOverflow !== this.props.alwaysRenderOverflow ||
            prevProps.visibleItemRenderer !== this.props.visibleItemRenderer) {
            // reset visible state if the above props change.
            this.setState({
                chopSize: this.defaultChopSize(),
                lastChopSize: null,
                lastOverflowCount: 0,
                overflow: [],
                repartitioning: true,
                visible: this.props.items,
            });
        }
        var _c = this.state, repartitioning = _c.repartitioning, overflow = _c.overflow, lastOverflowCount = _c.lastOverflowCount;
        if (
        // if a resize operation has just completed
        repartitioning === false &&
            prevState.repartitioning === true) {
            // only invoke the callback if the UI has actually changed
            if (overflow.length !== lastOverflowCount) {
                (_b = (_a = this.props).onOverflow) === null || _b === void 0 ? void 0 : _b.call(_a, overflow.slice());
            }
        }
        else if (!shallowCompareKeys(prevState, this.state)) {
            this.repartition();
        }
    };
    OverflowList.prototype.render = function () {
        var _this = this;
        var _a = this.props, className = _a.className, collapseFrom = _a.collapseFrom, observeParents = _a.observeParents, style = _a.style, _b = _a.tagName, tagName = _b === void 0 ? "div" : _b, visibleItemRenderer = _a.visibleItemRenderer;
        var overflow = this.maybeRenderOverflow();
        var list = React.createElement(tagName, {
            className: classNames(Classes.OVERFLOW_LIST, className),
            style: style,
        }, collapseFrom === Boundary.START ? overflow : null, this.state.visible.map(visibleItemRenderer), collapseFrom === Boundary.END ? overflow : null, React.createElement("div", { className: Classes.OVERFLOW_LIST_SPACER, ref: function (ref) { return (_this.spacer = ref); } }));
        return (React.createElement(ResizeSensor, { onResize: this.resize, observeParents: observeParents }, list));
    };
    OverflowList.prototype.maybeRenderOverflow = function () {
        var overflow = this.state.overflow;
        if (overflow.length === 0 && !this.props.alwaysRenderOverflow) {
            return null;
        }
        return this.props.overflowRenderer(overflow.slice());
    };
    OverflowList.prototype.repartition = function () {
        var _this = this;
        var _a;
        if (this.spacer == null) {
            return;
        }
        // if lastChopSize was 1, then our binary search has exhausted.
        var partitionExhausted = this.state.lastChopSize === 1;
        var minVisible = (_a = this.props.minVisibleItems) !== null && _a !== void 0 ? _a : 0;
        // spacer has flex-shrink and width 1px so if it's much smaller then we know to shrink
        var shouldShrink = this.spacer.offsetWidth < 0.9 && this.state.visible.length > minVisible;
        // we only check partitionExhausted for shouldGrow to ensure shrinking is the final operation.
        var shouldGrow = (this.spacer.offsetWidth >= 1 || this.state.visible.length < minVisible) &&
            this.state.overflow.length > 0 &&
            !partitionExhausted;
        if (shouldShrink || shouldGrow) {
            this.setState(function (state) {
                var visible;
                var overflow;
                if (_this.props.collapseFrom === Boundary.END) {
                    var result = shiftElements(state.visible, state.overflow, _this.state.chopSize * (shouldShrink ? 1 : -1));
                    visible = result[0];
                    overflow = result[1];
                }
                else {
                    var result = shiftElements(state.overflow, state.visible, _this.state.chopSize * (shouldShrink ? -1 : 1));
                    overflow = result[0];
                    visible = result[1];
                }
                return {
                    chopSize: halve(state.chopSize),
                    lastChopSize: state.chopSize,
                    // if we're starting a new partition cycle, record the last overflow count so we can track whether the UI changes after the new overflow is calculated
                    lastOverflowCount: _this.isFirstPartitionCycle(state.chopSize)
                        ? state.overflow.length
                        : state.lastOverflowCount,
                    overflow: overflow,
                    repartitioning: true,
                    visible: visible,
                };
            });
        }
        else {
            // repartition complete!
            this.setState({
                chopSize: this.defaultChopSize(),
                lastChopSize: null,
                repartitioning: false,
            });
        }
    };
    OverflowList.prototype.defaultChopSize = function () {
        return halve(this.props.items.length);
    };
    OverflowList.prototype.isFirstPartitionCycle = function (currentChopSize) {
        return currentChopSize === this.defaultChopSize();
    };
    OverflowList.displayName = "".concat(DISPLAYNAME_PREFIX, ".OverflowList");
    OverflowList.defaultProps = {
        alwaysRenderOverflow: false,
        collapseFrom: Boundary.START,
        minVisibleItems: 0,
    };
    return OverflowList;
}(React.Component));
export { OverflowList };
function halve(num) {
    return Math.ceil(num / 2);
}
function shiftElements(leftArray, rightArray, num) {
    // if num is positive then elements are shifted from left-to-right, if negative then right-to-left
    var allElements = leftArray.concat(rightArray);
    var newLeftLength = leftArray.length - num;
    if (newLeftLength <= 0) {
        return [[], allElements];
    }
    else if (newLeftLength >= allElements.length) {
        return [allElements, []];
    }
    var sliceIndex = allElements.length - newLeftLength;
    return [allElements.slice(0, -sliceIndex), allElements.slice(-sliceIndex)];
}
//# sourceMappingURL=overflowList.js.map