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
exports.PortalProvider = exports.PortalContext = void 0;
var tslib_1 = require("tslib");
var React = tslib_1.__importStar(require("react"));
/**
 * A React context to set options for all portals in a given subtree.
 * Do not use this PortalContext directly, instead use PortalProvider to set the options.
 */
exports.PortalContext = React.createContext({});
/**
 * Portal context provider.
 *
 * @see https://blueprintjs.com/docs/#core/context/portal-provider
 */
var PortalProvider = function (_a) {
    var children = _a.children, portalClassName = _a.portalClassName, portalContainer = _a.portalContainer;
    var contextOptions = React.useMemo(function () { return ({
        portalClassName: portalClassName,
        portalContainer: portalContainer,
    }); }, [portalClassName, portalContainer]);
    return React.createElement(exports.PortalContext.Provider, { value: contextOptions }, children);
};
exports.PortalProvider = PortalProvider;
//# sourceMappingURL=portalProvider.js.map