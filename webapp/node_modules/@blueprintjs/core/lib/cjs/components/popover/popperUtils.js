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
exports.getTransformOrigin = exports.getAlignment = exports.getOppositePlacement = exports.isVerticalPlacement = exports.getBasePlacement = exports.PopperPlacements = void 0;
var core_1 = require("@popperjs/core");
Object.defineProperty(exports, "PopperPlacements", { enumerable: true, get: function () { return core_1.placements; } });
// Popper placement utils
// ======================
/** Converts a full placement to one of the four positions by stripping text after the `-`. */
function getBasePlacement(placement) {
    return placement.split("-")[0];
}
exports.getBasePlacement = getBasePlacement;
/** Returns true if position is left or right. */
function isVerticalPlacement(side) {
    return ["left", "right"].indexOf(side) !== -1;
}
exports.isVerticalPlacement = isVerticalPlacement;
/** Returns the opposite position. */
function getOppositePlacement(side) {
    switch (side) {
        case "top":
            return "bottom";
        case "left":
            return "right";
        case "bottom":
            return "top";
        default:
            return "left";
    }
}
exports.getOppositePlacement = getOppositePlacement;
/** Returns the CSS alignment keyword corresponding to given placement. */
function getAlignment(placement) {
    var align = placement.split("-")[1];
    switch (align) {
        case "start":
            return "left";
        case "end":
            return "right";
        default:
            return "center";
    }
}
exports.getAlignment = getAlignment;
// Popper modifiers
// ================
/** Modifier helper function to compute popper transform-origin based on arrow position */
function getTransformOrigin(placement, arrowStyles) {
    var basePlacement = getBasePlacement(placement);
    if (arrowStyles === undefined) {
        return isVerticalPlacement(basePlacement)
            ? "".concat(getOppositePlacement(basePlacement), " ").concat(getAlignment(basePlacement))
            : "".concat(getAlignment(basePlacement), " ").concat(getOppositePlacement(basePlacement));
    }
    else {
        // const arrowSizeShift = state.elements.arrow.clientHeight / 2;
        var arrowSizeShift = 30 / 2;
        // can use keyword for dimension without the arrow, to ease computation burden.
        // move origin by half arrow's height to keep it centered.
        return isVerticalPlacement(basePlacement)
            ? "".concat(getOppositePlacement(basePlacement), " ").concat(parseInt(arrowStyles.top, 10) + arrowSizeShift, "px")
            : "".concat(parseInt(arrowStyles.left, 10) + arrowSizeShift, "px ").concat(getOppositePlacement(basePlacement));
    }
}
exports.getTransformOrigin = getTransformOrigin;
//# sourceMappingURL=popperUtils.js.map