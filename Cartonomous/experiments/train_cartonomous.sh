#!/usr/bin/env sh
set -e

caffe train --solver=cartonomous_solver.prototxt $@
