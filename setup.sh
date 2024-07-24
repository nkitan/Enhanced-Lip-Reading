#!/bin/sh

echo "Setting up Enhanced-Lip-Reading"
python3 -m venv env
while getopts 'i' OPTION; do
	case "$OPTION" in
		i) echo "Installing Dependencies"
			./env/bin/pip install -r requirements.txt 
			;;
	esac
done
echo "successfully installed Enhanced-Lip-Reading at $(pwd)"
./env/bin/python manage.py runserver &
