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
exports.iconNameToPathsRecordKey = exports.getIconPaths = exports.IconSvgPaths20 = exports.IconSvgPaths16 = void 0;
var tslib_1 = require("tslib");
var change_case_1 = require("change-case");
var IconSvgPaths16 = tslib_1.__importStar(require("./generated/16px/paths"));
exports.IconSvgPaths16 = IconSvgPaths16;
var IconSvgPaths20 = tslib_1.__importStar(require("./generated/20px/paths"));
exports.IconSvgPaths20 = IconSvgPaths20;
var iconTypes_1 = require("./iconTypes");
/**
 * Get the list of vector paths that define a given icon. These path strings are used to render `<path>`
 * elements inside an `<svg>` icon element. For full implementation details and nuances, see the icon component
 * handlebars template and `generate-icon-components` script in the __@blueprintjs/icons__ package.
 *
 * Note: this function loads all icon definitions __statically__, which means every icon is included in your
 * JS bundle. Only use this API if your app is likely to use all Blueprint icons at runtime. If you are looking for a
 * dynamic icon loader which loads icon definitions on-demand, use `{ Icons } from "@blueprintjs/icons"` instead.
 */
function getIconPaths(name, size) {
    var key = (0, change_case_1.pascalCase)(name);
    return size === iconTypes_1.IconSize.STANDARD ? IconSvgPaths16[key] : IconSvgPaths20[key];
}
exports.getIconPaths = getIconPaths;
/**
 * Type safe string literal conversion of snake-case icon names to PascalCase icon names.
 * This is useful for indexing into the SVG paths record to extract a single icon's SVG path definition.
 *
 * @deprecated use `getIconPaths` instead
 */
function iconNameToPathsRecordKey(name) {
    return (0, change_case_1.pascalCase)(name);
}
exports.iconNameToPathsRecordKey = iconNameToPathsRecordKey;
//# sourceMappingURL=allPaths.js.map