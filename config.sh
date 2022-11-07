#!/bin/bash
#
# Copyright (c) 2017-2022, King Abdullah University of Science and Technology
# ***************************************************************************
# *****      KAUST Extreme Computing Research Center Property           *****
# ***************************************************************************
#
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause. See the accompanying LICENSE file.
#
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'
PROJECT_SOURCE_DIR=$(dirname "$0")
ABSOLUE_PATH=$(dirname $(realpath "$0"))

while getopts ":tevhi:cd" opt; do
  case $opt in
  t) ##### Building tests enabled #####
    echo -e "${GREEN}Building tests enabled${NC}"
    BUILDING_TESTS="ON"
    ;;
  e) ##### Building examples enabled #####
    echo -e "${GREEN}Building examples enabled${NC}"
    BUILDING_EXAMPLES="ON"
    ;;
  i) ##### Define installation path  #####
    echo -e "${BLUE}Installation path set to $OPTARG${NC}"
    INSTALL_PATH=$OPTARG
    ;;
  v) ##### printing full output of make #####
    echo -e "${YELLOW}printing make with details${NC}"
    VERBOSE=ON
    ;;
  c)##### Using cuda enabled #####
    echo -e "${YELLOW}Cuda enabled ${NC}"
    USE_CUDA=ON
    ;;
  d)##### Using debug mode to build #####
    echo -e "${RED}Debug mode enabled ${NC}"
    BUILD_TYPE="DEBUG"
    ;;
  \?) ##### using default settings #####
    echo -e "${RED}Building tests disabled${NC}"
    echo -e "${RED}Building examples disabled${NC}"
    echo -e "${BLUE}Installation path set to /usr/local${NC}"
    echo -e "${RED}Using CUDA disabled${NC}"
    echo -e "${GREEN}Building in release mode${NC}"
    BUILDING_EXAMPLES="OFF"
    BUILDING_TESTS="OFF"
    INSTALL_PATH="/usr/local"
    VERBOSE=OFF
    USE_CUDA="OFF"
    BUILD_TYPE="RELEASE"
    ;;
  :) ##### Error in an option #####
    echo "Option $OPTARG requires parameter(s)"
    exit 0
    ;;
  h) ##### Prints the help #####
    echo "Usage of $(basename "$0"):"
    echo ""
    printf "%20s %s\n" "-t :" "to enable building tests"
    echo ""
    printf "%20s %s\n" "-e :" "to enable building examples"
    echo ""
    printf "%20s %s\n" "-c :" "to enable cuda support on"
    echo ""
    printf "%20s %s\n" "-d :" "to build in debug mode"
    echo ""
    printf "%20s %s\n" "-i [path] :" "specify installation path"
    printf "%20s %s\n" "" "default = /usr/local"
    echo ""
    exit 1
    ;;
  esac
done

if [ -z "$BUILDING_TESTS" ]; then
  BUILDING_TESTS="OFF"
  echo -e "${RED}Building tests disabled${NC}"
fi

if [ -z "$BUILDING_EXAMPLES" ]; then
  BUILDING_EXAMPLES="OFF"
  echo -e "${RED}Building examples disabled${NC}"
fi

if [ -z "$INSTALL_PATH" ]; then
  INSTALL_PATH="/usr/local"
  echo -e "${BLUE}Installation path set to $INSTALL_PATH${NC}"
fi

if [ -z "$USE_CUDA" ]; then
  USE_CUDA="OFF"
  echo -e "${RED}Using CUDA disabled${NC}"
fi

if [ -z "$BUILD_TYPE" ]; then
  BUILD_TYPE="RELEASE"
  echo -e "${GREEN}Building in release mode${NC}"
fi

rm -rf bin/
mkdir bin/
cmake -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  -DHCOREPP_BUILD_TESTS=$BUILDING_TESTS \
  -DHCOREPP_BUILD_EXAMPLES=$BUILDING_EXAMPLES \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PATH \
  -DCMAKE_VERBOSE_MAKEFILE:BOOL=$VERBOSE \
  -H"${PROJECT_SOURCE_DIR}" \
  -B"${PROJECT_SOURCE_DIR}/bin"\
  -DUSE_CUDA="$USE_CUDA"

