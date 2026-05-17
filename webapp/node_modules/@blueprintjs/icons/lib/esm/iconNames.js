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
import { __assign } from "tslib";
/* eslint-disable camelcase */
import { pascalCase, snakeCase } from "change-case";
// The two icon sets are identical aside from SVG paths, so we only need to import info for the 16px set
import { BlueprintIcons_16 } from "./generated/16px/blueprint-icons-16";
var IconNamesNew = {};
var IconNamesLegacy = {};
for (var _i = 0, _a = Object.values(BlueprintIcons_16); _i < _a.length; _i++) {
    var name_1 = _a[_i];
    IconNamesNew[pascalCase(name_1)] = name_1;
    IconNamesLegacy[snakeCase(name_1).toUpperCase()] = name_1;
}
export var IconNames = __assign(__assign({}, IconNamesNew), IconNamesLegacy);
export var IconNamesSet = new Set(Object.values(IconNames));
//# sourceMappingURL=iconNames.js.map