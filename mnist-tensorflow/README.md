# mnist-tensorflow
Specifies a Flyte workflow using TensorFlow on MNIST dataset.

## Testing using Flyte Sandbox environment
Start a flyte sandbox cluster using `flytectl` CLI mounting this directory to the sandbox Docker container.

```
flytectl sandbox start --source .
```

Build the Docker image used to run the example.

```
flytectl sandbox exec -- ./docker_build_and_tag.sh
```

Obtain the image tag, i.e., short git sha, and package the workflow.

```
pyflyte --pkgs flyte.workflows package --image "flyte:$GIT_SHA"
```

Upload the package to the Flyte backend, also known as registration. If you have not created a project yet do so.

```
flytectl create project --name mnist-tensorflow --id mnist-tensorflow
```

```
flytectl register files --project mnist-tensorflow --domain development --archive flyte-package.tgz --version v1
```

Head over to the Flyte UI and start the workflow.