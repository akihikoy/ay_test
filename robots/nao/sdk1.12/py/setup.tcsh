# Path of NAO SDK
setenv NAO_SDK_DIR /home/akihiko/prg/aldebaran/naoqi-sdk-1.12-linux32

setenv LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:${NAO_SDK_DIR}/lib"
# setenv PYTHONHOME "${NAO_SDK_DIR}"
setenv PYTHONPATH "${PYTHONPATH}:${NAO_SDK_DIR}/lib"
