/**
 * Wraps an async task with a performance timer. Only logs to console in development.
 */
export declare function wrapWithTimer(taskDescription: string, task: () => Promise<void>): Promise<void>;
