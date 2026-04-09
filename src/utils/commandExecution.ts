export type CommandExecutionAction<TActionId extends string = string> = {
  id: TActionId;
  args: Record<string, unknown>;
};

export type CommandExecutionPlan<TActionId extends string = string> = {
  actions: CommandExecutionAction<TActionId>[];
};

export interface CommandExecutionCallbacks<TActionId extends string = string> {
  onActionStart?: (action: CommandExecutionAction<TActionId>, index: number) => void;
  onActionSuccess?: (
    action: CommandExecutionAction<TActionId>,
    index: number,
    detail?: string
  ) => void;
  onActionFailure?: (
    action: CommandExecutionAction<TActionId>,
    index: number,
    error: unknown
  ) => void;
}

export interface CommandExecutionOptions<TContext, TActionId extends string = string>
  extends CommandExecutionCallbacks<TActionId> {
  context: TContext;
  executeAction: (
    action: CommandExecutionAction<TActionId>,
    context: TContext,
    index: number
  ) => Promise<string | void>;
}

export async function executeCommandPlanSequentially<TContext, TActionId extends string = string>(
  plan: CommandExecutionPlan<TActionId>,
  options: CommandExecutionOptions<TContext, TActionId>
) {
  for (const [index, action] of plan.actions.entries()) {
    options.onActionStart?.(action, index);

    try {
      const detail = await options.executeAction(action, options.context, index);
      options.onActionSuccess?.(action, index, detail ?? undefined);
    } catch (error) {
      options.onActionFailure?.(action, index, error);
      throw error;
    }
  }
}
