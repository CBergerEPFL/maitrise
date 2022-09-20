if [ -n "$PYTHONPATH" ]; then
    export PYTHONPATH='/root/.local/lib/python3.10/site-packages/pdm/pep582':$PYTHONPATH
else
    export PYTHONPATH='/root/.local/lib/python3.10/site-packages/pdm/pep582'
fi
