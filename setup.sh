#!/bin/sh

echo "Setting up Enhanced-Lip-Reading"
python3 -m venv env
source ./env/bin/activate
while getopts 'i' OPTION; do
	case "$OPTION" in
		i) echo "Installing Dependencies"
			pip install -r requirements.txt 
			;;
	esac
done
echo "successfully installed Enhanced-Lip-Reading at $(pwd)"
python manage.py runserver &
