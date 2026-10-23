"""Wait for the specific launched process, then run its reporting stage."""
import time,subprocess,sys
from pathlib import Path
pid=int(sys.argv[1])
while Path(f'/proc/{pid}').exists():time.sleep(15)
if not Path('/root/autodl-fs/d4rt_training/ttpo_controlled_20261014/COMPLETE.json').exists():
 raise SystemExit('Generation did not finish successfully; inspect ttpo_controlled.log')
subprocess.run([sys.executable,'audit_ttpo_controlled.py'],check=True)
