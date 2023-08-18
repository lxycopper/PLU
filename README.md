
## Quick Start

Some bookkeeping needs to be done for the code, like removing the local paths and so on. We will update these shortly. 

All config files can be found in: `configs/OWOD`

Sample command on a 4 GPU machine:
```python
python tools/train_net.py --num-gpus 4 --config-file <Change to the appropriate config file> SOLVER.IMS_PER_BATCH 4 SOLVER.BASE_LR 0.005
```

Kindly run `replicate.sh` to replicate results from the models shared on the Google Drive. 

Kindly check `run.sh` file for a task workflow.




