import * as React from "react";
export type AsyncControllableTextAreaProps = React.TextareaHTMLAttributes<HTMLTextAreaElement>;
/**
 * A wrapper around the low-level <textarea> component which works around a React bug
 * the same way <AsyncControllableInput> does.
 */
export declare const AsyncControllableTextArea: React.ForwardRefExoticComponent<AsyncControllableTextAreaProps & React.RefAttributes<HTMLTextAreaElement>>;
