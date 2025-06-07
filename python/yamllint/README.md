# Install yamllint

```bash
$ sudo apt -f install yamllint
```

# Usage

NOTE:

- The reason of putting "%YAML:1.0" at the beginning is to make a YAML compatible with OpenCV.

## Using yamllint from command line:

```bash
$ sed '1s/^%YAML:1\.0$/ /' sample.yaml | yamllint -c yamllint.yaml -
```

## From the script with extended warning checks

```bash
$ ./yamllint_ext.py sample.yaml
```

