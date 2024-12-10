srun --job-name="langchain" \
     --time=4:00:00 \
     --account=bcsn-hydro \
     --partition=a100 \
     --gres=gpu:1 \
     --nodes=1 \
     --mem=200G \
     --cpus-per-task=32 \
     --ntasks-per-node=1 \
     --pty bash



srun --job-name="binomial log" \
     --time=48:00:00 \
     --account=bcsn-hydro \
     --partition=interlagos \
     --nodes=1 \
     --mem=480G \
     --cpus-per-task=64 \
     --ntasks-per-node=1 \
     --pty bash