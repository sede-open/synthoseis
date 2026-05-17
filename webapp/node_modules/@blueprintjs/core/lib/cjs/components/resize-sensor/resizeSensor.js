"use strict";
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
Object.defineProperty(exports, "__esModule", { value: true });
exports.ResizeSensor = void 0;
var tslib_1 = require("tslib");
var React = tslib_1.__importStar(require("react"));
var common_1 = require("../../common");
/**
 * Resize sensor component.
 *
 * It requires a single DOM element child and will error otherwise.
 *
 * @see https://blueprintjs.com/docs/#core/components/resize-sensor
 **/
var ResizeSensor = /** @class */ (function (_super) {
    tslib_1.__extends(ResizeSensor, _super);
    function ResizeSensor() {
        var _a;
        var _this = _super.apply(this, arguments) || this;
        _this.targetRef = (_a = _this.props.targetRef) !== null && _a !== void 0 ? _a : React.createRef();
        _this.prevElement = undefined;
        return _this;
    }
    ResizeSensor.prototype.render = function () {
        var onlyChild = React.Children.only(this.props.children);
        // If we're provided a mutable ref to the child element already, we must re-use that one. This is necessary
        // in cases where the child node is not a native DOM element and does not use `React.forwardRef`, since
        // there's no way for us to know how to attach to the underlying DOM node.
        if (this.props.targetRef !== undefined) {
            return onlyChild;
        }
        return React.cloneElement(onlyChild, { ref: this.targetRef });
    };
    ResizeSensor.prototype.componentDidMount = function () {
        var _this = this;
        // ResizeObserver is available in all modern browsers supported by Blueprint but not in server-side rendering
        // and some test environments like jsdom, so we to do a feature check here.
        this.observer =
            globalThis.ResizeObserver != null
                ? new ResizeObserver(function (entries) { var _a, _b; return (_b = (_a = _this.props).onResize) === null || _b === void 0 ? void 0 : _b.call(_a, entries); })
                : undefined;
        this.observeElement();
    };
    ResizeSensor.prototype.componentDidUpdate = function (prevProps) {
        this.observeElement(this.props.observeParents !== prevProps.observeParents);
    };
    ResizeSensor.prototype.componentWillUnmount = function () {
        var _a;
        (_a = this.observer) === null || _a === void 0 ? void 0 : _a.disconnect();
        this.prevElement = undefined;
    };
    /**
     * Observe the DOM element, if defined and different from the currently
     * observed element. Pass `force` argument to skip element checks and always
     * re-observe.
     */
    ResizeSensor.prototype.observeElement = function (force) {
        if (force === void 0) { force = false; }
        if (this.observer === undefined) {
            return;
        }
        if (!(this.targetRef.current instanceof Element)) {
            // stop everything if not defined
            this.observer.disconnect();
            return;
        }
        if (this.targetRef.current === this.prevElement && !force) {
            // quit if given same element -- nothing to update (unless forced)
            return;
        }
        else {
            // clear observer list if new element
            this.observer.disconnect();
            // remember element reference for next time
            this.prevElement = this.targetRef.current;
        }
        // observer callback is invoked immediately when observing new elements
        this.observer.observe(this.targetRef.current);
        if (this.props.observeParents) {
            var parent_1 = this.targetRef.current.parentElement;
            while (parent_1 != null) {
                this.observer.observe(parent_1);
                parent_1 = parent_1.parentElement;
            }
        }
    };
    ResizeSensor.displayName = "".concat(common_1.DISPLAYNAME_PREFIX, ".ResizeSensor");
    return ResizeSensor;
}(common_1.AbstractPureComponent));
exports.ResizeSensor = ResizeSensor;
//# sourceMappingURL=resizeSensor.js.map