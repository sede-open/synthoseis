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
exports.splitPathsBySizeLoader = void 0;
var tslib_1 = require("tslib");
var change_case_1 = require("change-case");
var iconTypes_1 = require("../iconTypes");
/**
 * A dynamic loader for icon paths that generates separate chunks for the two size variants.
 */
var splitPathsBySizeLoader = function (name, size) { return tslib_1.__awaiter(void 0, void 0, void 0, function () {
    var key, pathsRecord;
    return tslib_1.__generator(this, function (_a) {
        switch (_a.label) {
            case 0:
                key = (0, change_case_1.pascalCase)(name);
                if (!(size === iconTypes_1.IconSize.STANDARD)) return [3 /*break*/, 2];
                return [4 /*yield*/, Promise.resolve().then(function () { return tslib_1.__importStar(require(
                    /* webpackChunkName: "blueprint-icons-16px-paths" */
                    "../generated/16px/paths")); })];
            case 1:
                pathsRecord = _a.sent();
                return [3 /*break*/, 4];
            case 2: return [4 /*yield*/, Promise.resolve().then(function () { return tslib_1.__importStar(require(
                /* webpackChunkName: "blueprint-icons-20px-paths" */
                "../generated/20px/paths")); })];
            case 3:
                pathsRecord = _a.sent();
                _a.label = 4;
            case 4: return [2 /*return*/, pathsRecord[key]];
        }
    });
}); };
exports.splitPathsBySizeLoader = splitPathsBySizeLoader;
//# sourceMappingURL=splitPathsBySizeLoader.js.map