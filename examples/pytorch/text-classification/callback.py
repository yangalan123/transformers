from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
import os, json
class GradValueClippingCallback(TrainerCallback):
    "A callback that helps deals with initialize memory to save grad value clip results and dump it at the end of training"
    # def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
    #     if hasattr(self, "gradclipmemory"):
    #         self.gradClipMemory.clear()
    #     else:
    #         self.gradClipMemory = {}
    #     self.gradClipMemorySavePath = os.path.join(args.output_dir, "gradClipMemoryJsons")

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if (state.global_step + 1) % args.grad_clip_data_save_period == 0:
            # self.gradClipMemory["step"] = state.global_step
            if hasattr(state, "gradClipMemory"):
                state.gradClipMemory.clear()
            else:
                state.gradClipMemory = {}
            state.gradClipMemory["step"] = state.global_step


    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if not os.path.exists(args.gradClipMemorySavePath):
            os.makedirs(args.gradClipMemorySavePath)
        if hasattr(state, "gradClipMemory") and len(state.gradClipMemory) > 0:
            with open(os.path.join(args.gradClipMemorySavePath, f"status_{state.global_step}.json"), "w", encoding='utf-8') as f_out:
                json.dump(state.gradClipMemory, f_out)
                state.gradClipMemory.clear()
