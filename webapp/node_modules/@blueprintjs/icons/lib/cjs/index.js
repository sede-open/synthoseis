"use strict";
/*
 * Copyright 2022 Palantir Technologies, Inc. All rights reserved.
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
exports.IconSize = exports.IconNames = exports.IconCodepoints = exports.getIconContentString = exports.SVGIconContainer = exports.Icons = exports.getIconPaths = exports.IconSvgPaths20 = exports.IconSvgPaths16 = void 0;
// N.B. these named imports will trigger bundlers to statically loads all icon path modules
var allPaths_1 = require("./allPaths");
Object.defineProperty(exports, "IconSvgPaths16", { enumerable: true, get: function () { return allPaths_1.IconSvgPaths16; } });
Object.defineProperty(exports, "IconSvgPaths20", { enumerable: true, get: function () { return allPaths_1.IconSvgPaths20; } });
Object.defineProperty(exports, "getIconPaths", { enumerable: true, get: function () { return allPaths_1.getIconPaths; } });
var iconLoader_1 = require("./iconLoader");
Object.defineProperty(exports, "Icons", { enumerable: true, get: function () { return iconLoader_1.Icons; } });
var svgIconContainer_1 = require("./svgIconContainer");
Object.defineProperty(exports, "SVGIconContainer", { enumerable: true, get: function () { return svgIconContainer_1.SVGIconContainer; } });
var iconCodepoints_1 = require("./iconCodepoints");
Object.defineProperty(exports, "getIconContentString", { enumerable: true, get: function () { return iconCodepoints_1.getIconContentString; } });
Object.defineProperty(exports, "IconCodepoints", { enumerable: true, get: function () { return iconCodepoints_1.IconCodepoints; } });
var iconNames_1 = require("./iconNames");
Object.defineProperty(exports, "IconNames", { enumerable: true, get: function () { return iconNames_1.IconNames; } });
var iconTypes_1 = require("./iconTypes");
Object.defineProperty(exports, "IconSize", { enumerable: true, get: function () { return iconTypes_1.IconSize; } });
//# sourceMappingURL=index.js.map