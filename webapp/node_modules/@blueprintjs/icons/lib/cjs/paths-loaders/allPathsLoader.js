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
exports.allPathsLoader = void 0;
var tslib_1 = require("tslib");
/**
 * A simple module loader which concatenates all icon paths into a single chunk.
 */
var allPathsLoader = function (name, size) { return tslib_1.__awaiter(void 0, void 0, void 0, function () {
    var getIconPaths;
    return tslib_1.__generator(this, function (_a) {
        switch (_a.label) {
            case 0: return [4 /*yield*/, Promise.resolve().then(function () { return tslib_1.__importStar(require(
                /* webpackChunkName: "blueprint-icons-all-paths" */
                "../allPaths")); })];
            case 1:
                getIconPaths = (_a.sent()).getIconPaths;
                return [2 /*return*/, getIconPaths(name, size)];
        }
    });
}); };
exports.allPathsLoader = allPathsLoader;
//# sourceMappingURL=allPathsLoader.js.map