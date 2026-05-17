/*
 * Copyright 2016 Palantir Technologies, Inc. All rights reserved.
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
import * as React from "react";
import { AbstractPureComponent, DISPLAYNAME_PREFIX, Intent } from "../../common";
import * as Errors from "../../common/errors";
import { MultiSlider } from "./multiSlider";
var RangeIndex;
(function (RangeIndex) {
    RangeIndex[RangeIndex["START"] = 0] = "START";
    RangeIndex[RangeIndex["END"] = 1] = "END";
})(RangeIndex || (RangeIndex = {}));
/**
 * Range slider component.
 *
 * @see https://blueprintjs.com/docs/#core/components/sliders.range-slider
 */
var RangeSlider = /** @class */ (function (_super) {
    __extends(RangeSlider, _super);
    function RangeSlider() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    RangeSlider.prototype.render = function () {
        var _a = this.props, value = _a.value, handleHtmlProps = _a.handleHtmlProps, props = __rest(_a, ["value", "handleHtmlProps"]);
        return (React.createElement(MultiSlider, __assign({}, props),
            React.createElement(MultiSlider.Handle, { value: value[RangeIndex.START], type: "start", intentAfter: props.intent, htmlProps: handleHtmlProps === null || handleHtmlProps === void 0 ? void 0 : handleHtmlProps.start }),
            React.createElement(MultiSlider.Handle, { value: value[RangeIndex.END], type: "end", htmlProps: handleHtmlProps === null || handleHtmlProps === void 0 ? void 0 : handleHtmlProps.end })));
    };
    RangeSlider.prototype.validateProps = function (props) {
        var value = props.value;
        if (value == null || value[RangeIndex.START] == null || value[RangeIndex.END] == null) {
            throw new Error(Errors.RANGESLIDER_NULL_VALUE);
        }
    };
    RangeSlider.defaultProps = __assign(__assign({}, MultiSlider.defaultSliderProps), { intent: Intent.PRIMARY, value: [0, 10] });
    RangeSlider.displayName = "".concat(DISPLAYNAME_PREFIX, ".RangeSlider");
    return RangeSlider;
}(AbstractPureComponent));
export { RangeSlider };
//# sourceMappingURL=rangeSlider.js.map