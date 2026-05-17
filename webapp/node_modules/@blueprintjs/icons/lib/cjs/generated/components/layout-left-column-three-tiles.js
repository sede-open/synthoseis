"use strict";
/*
 * Copyright 2024 Palantir Technologies, Inc. All rights reserved.
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
exports.LayoutLeftColumnThreeTiles = void 0;
var tslib_1 = require("tslib");
var React = tslib_1.__importStar(require("react"));
var iconTypes_1 = require("../../iconTypes");
var svgIconContainer_1 = require("../../svgIconContainer");
exports.LayoutLeftColumnThreeTiles = React.forwardRef(function (props, ref) {
    var isLarge = props.size >= iconTypes_1.IconSize.LARGE;
    var pixelGridSize = isLarge ? iconTypes_1.IconSize.LARGE : iconTypes_1.IconSize.STANDARD;
    var translation = "".concat(-1 * pixelGridSize / 0.05 / 2);
    var style = { transformOrigin: "center" };
    return (React.createElement(svgIconContainer_1.SVGIconContainer, tslib_1.__assign({ iconName: "layout-left-column-three-tiles", ref: ref }, props),
        React.createElement("path", { d: isLarge ? "M220 380C220 391.04568 228.954 400 240 400H380C391.046 400 400 391.0457 400 380V20C400 8.954 391.046 0 380 0H240C228.954 0 220 8.954 220 20V380zM0 380C0 391.0457 8.9543 400 20 400H160C171.0456 400 180 391.0457 180 380V320C180 308.9544 171.0456 300 160 300H20C8.9543 300 0 308.9544 0 320V380zM0 240C0 251.0456 8.9543 260 20 260H160C171.0456 260 180 251.0456 180 240V160C180 148.954 171.0456 140 160 140H20C8.9543 140 0 148.954 0 160V240zM0 80C0 91.046 8.9543 100 20 100H160C171.0456 100 180 91.046 180 80V20C180 8.954 171.0456 0 160 0H20C8.9543 0 0 8.954 0 20V80z" : "M140 180C140 191.0456 131.0456 200 120 200H20C8.9543 200 0 191.0456 0 180V140C0 128.9544 8.9543 120 20 120H120C131.0456 120 140 128.9544 140 140V180zM140 300C140 311.0457 131.0456 320 120 320H20C8.9543 320 0 311.0457 0 300V260C0 248.9544 8.9543 240 20 240H120C131.0456 240 140 248.9544 140 260V300zM140 60C140 71.046 131.0456 80 120 80H20C8.9543 80 0 71.046 0 60V20C0 8.954 8.9543 0 20 0H120C131.0456 0 140 8.954 140 20V60zM320 300C320 311.0457 311.046 320 300 320H200C188.9544 320 180 311.0457 180 300V20C180 8.954 188.9544 0 200 0H300C311.046 0 320 8.954 320 20V300z", fillRule: "evenodd", transform: "scale(0.05, -0.05) translate(".concat(translation, ", ").concat(translation, ")"), style: style })));
});
exports.LayoutLeftColumnThreeTiles.defaultProps = {
    size: iconTypes_1.IconSize.STANDARD,
};
exports.LayoutLeftColumnThreeTiles.displayName = "Blueprint5.Icon.LayoutLeftColumnThreeTiles";
exports.default = exports.LayoutLeftColumnThreeTiles;
//# sourceMappingURL=layout-left-column-three-tiles.js.map