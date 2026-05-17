import * as React from "react";
import type { HotkeyConfig } from "./hotkeyConfig";
export interface UseHotkeysOptions {
    /**
     * A custom document to reference when binding global event handlers.
     * This can be useful when using iframes in an application.
     *
     * @default window.document
     */
    document?: Document;
    /**
     * The key combo which will trigger the hotkeys dialog to open.
     *
     * @default "?"
     */
    showDialogKeyCombo?: string;
}
export interface UseHotkeysReturnValue {
    handleKeyDown: React.KeyboardEventHandler<HTMLElement>;
    handleKeyUp: React.KeyboardEventHandler<HTMLElement>;
}
/**
 * React hook to register global and local hotkeys for a component.
 *
 * @see https://blueprintjs.com/docs/#core/hooks/use-hotkeys
 * @param keys list of hotkeys to configure
 * @param options hook options
 */
export declare function useHotkeys(keys: readonly HotkeyConfig[], options?: UseHotkeysOptions): UseHotkeysReturnValue;
