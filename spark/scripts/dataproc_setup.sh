#!/bin/bash
set -e

# General setup
#PORT="10000"   
PORT=$(( ( RANDOM % 1100 )  + 1000 ))
ZONE="us-central1-f"

# Do not change nothing bellow this
USAGE="Usage: `basename $0` <PROJECT> <CLUSTERNAME> <ACTION> (start|browse|stop) <WORKERS>"

# Read cmdline arguments
PROJECT=$1
CLUSTERNAME=$2
ACTION=$3
NUMERIC_ARG=$4

# Find out if we are using to browse or start cluster
if [ ! -z "$NUMERIC_ARG" ]; then
  if [ "$NUMERIC_ARG" -lt "1025" ]; then
    WORKERS=$NUMERIC_ARG
  else
    PORT=$NUMERIC_ARG
  fi
fi

if [ -z "$CLUSTERNAME" ]; then
  echo 'CLUSTERNAME argument is required.'
  echo $USAGE
  exit 1
fi

if [ -z "$ACTION" ]; then
  echo 'ACTION argument is required.'
  echo $USAGE
  exit 1
fi

if [ $ACTION != "start" ] && [ $ACTION != "browse" ] && [ $ACTION != "stop" ]; then
  echo 'Invalid action!'
  echo $USAGE
  exit 1
fi

if [ -z "$WORKERS" ]; then
  WORKERS=8
  #WORKERS=2
fi


BUCKET="$CLUSTERNAME-bucket"
echo "Using '$WORKERS' workers..."
echo "Using '$BUCKET' bucket..."
echo "ACTION '$ACTION' selected for cluster '$CLUSTERNAME' on '$PORT' port."

### END OF SETUP

### Creat bucket if needed
create_bucket() {
  if [ -z `gsutil ls | grep gs://$BUCKET/` ]
  then
    echo "Creating bucket $BUCKET..."
    gsutil mb -p $PROJECT gs://$BUCKET
  else
    echo "Bucket $BUCKET already exists. Reusing..."
  fi

  return 0
}


### Upload FILES
upload_files() {
  gsutil -m cp ./setup/* gs://$BUCKET/setup/
  return 0
}


### Create cluster
create_cluster() {
  gcloud dataproc clusters create $CLUSTERNAME \
  --image-version 1.1 \
  --project $PROJECT \
  --bucket $BUCKET \
  --zone $ZONE \
  --num-workers $WORKERS \
  --scopes cloud-platform \
  --initialization-actions gs://dataproc-initialization-actions/jupyter/jupyter.sh,gs://$BUCKET/setup/python_conf.sh,gs://$BUCKET/setup/linux_conf.sh \
  --initialization-action-timeout 40m \
  --master-machine-type "n1-highmem-4" \
  --worker-machine-type "n1-highmem-4" \
  --properties spark:spark.driver.maxResultSize=20328m,spark:spark.driver.memory=20656m,spark:spark.executor.heartbeatInterval=30s,spark:spark.yarn.executor.memoryOverhead=5058,spark:spark.executor.memory=9310m,spark:spark.yarn.am.memory=8586m

  return 0
}

### Backup files on cluster
backup_files() {
  gcloud compute --project $PROJECT copy-files --zone "us-central1-f" "$CLUSTERNAME-m:/tmp/*.ipynb" ../src/notebooks/
  gsutil -m cp ../src/notebooks/* gs://backup-notebooks/
}


### Open ssh tunnel
open_tunnel() {
  gcloud compute ssh --zone=$ZONE \
  --ssh-flag="-D" --ssh-flag="$PORT" --ssh-flag="-N" --ssh-flag="-n" --project "$PROJECT" "$CLUSTERNAME-m" &
  # gcloud compute ssh --zone $1 --ssh-flag="-D $PORT" --ssh-flag="-N" --ssh-flag="-n" --project $2 $3 & 
  return 0
}

### Open browser
open_browser() {
  google-chrome \
    "http://$CLUSTERNAME-m:8123" \
    --proxy-server="socks5://localhost:$PORT" \
    --host-resolver-rules="MAP * 0.0.0.0 , EXCLUDE localhost" \
    --user-data-dir=${HOME}/.google-chrome/session${DISPLAY}
}

if [ $ACTION = "start" ]; then
  create_bucket
  upload_files
  create_cluster

  exit 0
fi

if [ $ACTION = "browse" ]; then
  open_tunnel
  sleep 7
  open_browser
  trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT
  wait
  exit 0
fi

if [ $ACTION = "stop" ]; then
  gcloud dataproc clusters delete $CLUSTERNAME
  gsutil -m rm -r gs://$BUCKET/
  exit 0
fi

############ STOP
echo "Saindo"
exit 1
