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
exports.IconNamesSet = exports.IconNames = void 0;
var tslib_1 = require("tslib");
/* eslint-disable camelcase */
var change_case_1 = require("change-case");
// The two icon sets are identical aside from SVG paths, so we only need to import info for the 16px set
var blueprint_icons_16_1 = require("./generated/16px/blueprint-icons-16");
var IconNamesNew = {};
var IconNamesLegacy = {};
for (var _i = 0, _a = Object.values(blueprint_icons_16_1.BlueprintIcons_16); _i < _a.length; _i++) {
    var name_1 = _a[_i];
    IconNamesNew[(0, change_case_1.pascalCase)(name_1)] = name_1;
    IconNamesLegacy[(0, change_case_1.snakeCase)(name_1).toUpperCase()] = name_1;
}
exports.IconNames = tslib_1.__assign(tslib_1.__assign({}, IconNamesNew), IconNamesLegacy);
exports.IconNamesSet = new Set(Object.values(exports.IconNames));
//# sourceMappingURL=iconNames.js.map