#!/usr/bin/env python3

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vit.inference.evaluate import main  # noqa: E402

if __name__ == "__main__":
    main()
