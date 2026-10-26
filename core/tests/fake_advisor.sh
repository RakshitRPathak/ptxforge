#!/bin/sh
# Test double for the external (LLM) advisor protocol: ignores the context file, prints suggestions
# that include one valid config, one hallucinated config, and one malformed entry.
cat <<'EOF'
Sure! Here are my suggestions:
{"proposals":[
 {"BM":128,"BN":128,"BK":8,"TM":8,"TN":8,"PAD":4,"why":"bigger tile raises arithmetic intensity"},
 {"BM":512,"BN":64,"BK":8,"TM":4,"TN":4,"PAD":0,"why":"not in the search domain"},
 {"BM":64}
]}
EOF
