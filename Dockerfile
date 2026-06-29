# Start from the official ROS 2 Humble Desktop image
FROM osrf/ros:humble-desktop

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /root

# Install essential build tools and utilities
RUN apt-get update && apt-get install -y \
    curl wget git nano usbutils \
    && rm -rf /var/lib/apt/lists/*

# Download and execute the official Trossen Robotics installation script
# -d humble : Installs for ROS 2 Humble
# -n : Runs the script in non-interactive mode
RUN curl 'https://raw.githubusercontent.com/Interbotix/interbotix_ros_manipulators/main/interbotix_ros_xsarms/install/amd64/xsarm_amd64_install.sh' > xsarm_amd64_install.sh && \
    chmod +x xsarm_amd64_install.sh && \
    ./xsarm_amd64_install.sh -d humble -n

RUN pip3 install --no-cache-dir pyrealsense2

# Automatically source ROS 2 and the Interbotix workspace when opening a bash shell
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /root/interbotix_ws/install/setup.bash" >> ~/.bashrc

CMD ["bash"]
