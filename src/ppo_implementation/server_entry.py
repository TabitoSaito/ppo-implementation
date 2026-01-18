from tensorboard import program


logdir = "runs"
host = "0.0.0.0"
port = 5000

tb = program.TensorBoard()
tb.configure(
    argv=[
        None,
        "--logdir",
        logdir,
        "--host",
        host,
        "--port",
        str(port),
    ]
)

url = tb.launch()
print(f"TensorBoard läuft unter: {url}")

input()
