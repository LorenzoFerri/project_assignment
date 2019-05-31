## Lorenzo Ferri, Marco Nobile

# Distance and angle estimator with Mighty Thymio

In order to run you have first to train the cnn, we didn't push it to git since the file was over 500 MB.

To train the cnn you first need to generate the dataset. to do so launch gazebo with:

    roslaunch project_assignment thymio_gazebo_bringup.launch

Then call the script from inside the scripts folder:

    ./create_dataset.py thymio10

Then press space to start generating the dataset.

Now you can train the cnn running from the scripts folder:

    python3 main.py train

This will take a while. On a GTX 1070 it took around 30 min.

Now you can test the estimator running

    roslaunch project_assignment thymio_gazebo_bringup.launch
    ./move.py thymio10

Move using the keys:

    w = move forward
    x = move backward
    s = stop moving

    a = turn left
    d = turn right
    e = stop turning
