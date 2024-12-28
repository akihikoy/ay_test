#!/bin/bash
#\file    git_clone_or_pull.sh
#\brief   certain bash script
#\author  Akihiko Yamaguchi, info@akihikoy.net
#\version 0.1
#\date    Dec.28, 2024


## Clone or pull a git repository.
#
# Usage:
#   git_clone_or_pull [--branch <branch_name>] [--commit <commit_hash>] <repo_url>
#
# Behavior:
#   1) If the repository is not cloned yet,
#   1.1) clone it.
#   1.2) If a branch is specified, checkout that branch.
#   1.3) If a commit is specified, checkout that commit.
#   2) If the repository already exists,
#   2.1) if a commit is specified, do nothing.
#   2.2) Otherwise, pull it.
##
function git_clone_or_pull() {
  local BRANCH=""
  local COMMIT=""
  local REPO_URL=""

  # Parse arguments
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --branch)
        BRANCH="$2"
        shift 2
        ;;
      --commit)
        COMMIT="$2"
        shift 2
        ;;
      *)
        REPO_URL="$1"
        shift 1
        ;;
    esac
  done

  # Check if the repository URL is provided
  if [[ -z "$REPO_URL" ]]; then
    echo "[ERROR] Repository URL is not specified."
    return 1
  fi

  # Obtain the repository directory name (remove the trailing .git if present)
  local REPO_NAME
  REPO_NAME="$(basename "$REPO_URL" .git)"

  # Check if the repository has already been cloned
  if [[ -e "$REPO_NAME/.git" ]]; then
    # If a commit is specified and the repo already exists, do nothing
    if [[ -n "$COMMIT" ]]; then
      echo "[INFO] The repository already exists, and a commit was specified. Doing nothing: $REPO_NAME"
      return 0
    fi

    # If no commit is specified, pull the latest changes
    echo "[INFO] Pulling the existing repository: $REPO_NAME"
    (
      cd "$REPO_NAME" || exit 1
      git pull
    )

  else
    # Clone the repository
    echo "[INFO] Cloning the repository: $REPO_URL"
    git clone "$REPO_URL"
    if [[ $? -ne 0 ]]; then
      echo "[ERROR] Failed to clone the repository."
      return 1
    fi

    # If a branch is specified, checkout that branch
    if [[ -n "$BRANCH" ]]; then
      echo "[INFO] Checking out branch: $BRANCH"
      (
        cd "$REPO_NAME" || exit 1
        git checkout "$BRANCH"
      )
    fi

    # If a commit is specified and the repo was just cloned, checkout the commit
    # (If the repo already existed, we do nothing based on the requirement.)
    if [[ -n "$COMMIT" && -d "$REPO_NAME" ]]; then
      echo "[INFO] Checking out commit: $COMMIT"
      (
        cd "$REPO_NAME" || exit 1
        git checkout "$COMMIT"
      )
    fi

  fi

  echo "[INFO] Done: $REPO_NAME"
}



git_clone_or_pull --commit 9777868cd https://github.com/UniversalRobots/Universal_Robots_ROS_Driver.git
git_clone_or_pull --branch calibration_devel https://github.com/fmauch/universal_robot.git
git_clone_or_pull https://github.com/ROBOTIS-GIT/DynamixelSDK.git


