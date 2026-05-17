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
exports.getIconContentString = exports.IconCodepoints = void 0;
// The two icon sets are identical aside from SVG paths, so we only need to import info for the 16px set
var blueprint_icons_16_1 = require("./generated/16px/blueprint-icons-16");
/**
 * Icon codepoints as base 10 number strings. If you need to render these strings
 * into an SVG document or as `::before` pseudo content, consider using `getIconContentString()`
 * instead.
 */
exports.IconCodepoints = blueprint_icons_16_1.BLUEPRINT_ICONS_16_CODEPOINTS;
/**
 * Returns the hex code content string which represents the codepoint in the icon font
 * for a given icon. You can render this string to the DOM and if the icon font is loaded
 * as an active font family, this string will be replaced with the associated icon.
 */
function getIconContentString(icon) {
    // parse base 10 number from string, then convert to hex code
    return String.fromCodePoint(parseInt(exports.IconCodepoints[icon], 10));
}
exports.getIconContentString = getIconContentString;
//# sourceMappingURL=iconCodepoints.js.map