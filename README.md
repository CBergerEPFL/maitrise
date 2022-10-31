# ecg_quality_assesment

For optimal experience, use with VSCode's devcontainer and install recommended extensions

## Instructions

### Adding new library
- run `pdm add numpy`

### Adding new dev library
- run `pdm add -d numpy`

### Adding library from local "temporary_source"
- copy libraryname-version-py3-none-any.whl file to the temporary_source folder and run
- run `pdm add ./temporary_source/libraryname-version-py3-none-any.whl`

### Building library
- update version in pyproject.toml>version
- run `pdm build`
