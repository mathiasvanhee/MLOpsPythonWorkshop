resource_group=$1
workspace_name=$2
dataset_name=waldo
dataset_version=1

echo "Init Waldone dataset"

RESULT=$(az ml data show --resource-group $resource_group --workspace-name $workspace_name --version $dataset_version --name $dataset_name > /dev/null 2>&1; echo $?)
if [ $RESULT -eq 0 ]; then
  echo "Dataset already exist"
else
  echo "Dataset not found so create it"
  git clone https://github.com/maxenceTroislouche/mlops-waldone-dataset.git
  rm -rf mlops-waldone-dataset/.git
  az ml data create -f waldone-dataset.yml --resource-group $resource_group --workspace-name $workspace_name
fi

echo "Init Waldone dataset done"
