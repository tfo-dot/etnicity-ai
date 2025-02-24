pip install poetry

poetry add --group dev jupyterlab ipykernel

poetry install

poetry run python -m ipykernel install --user --name=poetry-root-env

poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

if (!(Test-Path -Path "./data")) {
  New-Item -ItemType Directory -Path "./data"

  Set-Location -Path "data"

  poetry run gdown --id 1n1wjPXLRlMrJjFdX1SxoiaO1JgwbQuFV

  Expand-Archive -Path .\fairface-img-margin025-trainval.zip -DestinationPath .\images

  poetry run gdown --id 1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH

  poetry run gdown --id 1wOdja-ezstMEp81tX1a-EYkFebev4h7D

  Set-Location -Path ".."
} else {
  Write-Host "Skipping data download"
}

poetry run pip install -e .

Set-Location -Path "notebooks"

jupyter-lab.exe

Set-Location -Path ".."