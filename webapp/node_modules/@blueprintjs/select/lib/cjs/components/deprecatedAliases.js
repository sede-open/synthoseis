"use strict";
/*
 * Copyright 2023 Palantir Technologies, Inc. All rights reserved.
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
exports.Suggest2 = exports.Select2 = exports.MultiSelect2 = void 0;
var multiSelect_1 = require("./multi-select/multiSelect");
/** @deprecated import "v1" API instead */
Object.defineProperty(exports, "MultiSelect2", { enumerable: true, get: function () { return multiSelect_1.MultiSelect; } });
var select_1 = require("./select/select");
/** @deprecated import "v1" API instead */
Object.defineProperty(exports, "Select2", { enumerable: true, get: function () { return select_1.Select; } });
var suggest_1 = require("./suggest/suggest");
/** @deprecated import "v1" API instead */
Object.defineProperty(exports, "Suggest2", { enumerable: true, get: function () { return suggest_1.Suggest; } });
//# sourceMappingURL=deprecatedAliases.js.map