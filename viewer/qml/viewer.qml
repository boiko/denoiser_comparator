import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.2

Rectangle {
    id: root

    function image_path(name, suffix) {
        return result_data.image_path + "/" + name + "_" + suffix + ".png"
    }

    color: "#eeeeeeff"

    Rectangle {
        id: imageListContainer
        width: 200
        anchors {
            top: parent.top
            bottom: parent.bottom
            left: parent.left
            margins: 10
        }
        border.color: "black"

        ListView {
            id: imageList

            property string currentImage: model[currentIndex]

            clip: true
            focus: true
            anchors {
                fill: parent
                margins: 1
            }
            highlight: Rectangle {
                color: "lightBlue"
            }
            highlightFollowsCurrentItem: true
            keyNavigationEnabled: true

            header: Rectangle {
                gradient: Gradient {
                    GradientStop { position: 0.0; color: "lightGray" }
                    GradientStop { position: 1.0; color: "#eeeeeeff" }
                }

                anchors {
                    top:  parent.top
                    left: parent.left
                    right: parent.right
                }
                height: listLabel.contentHeight * 1.3

                Text {
                    id: listLabel
                    text: "Image:"
                    anchors {
                        fill: parent
                        margins: 5
                    }
                    verticalAlignment: Text.AlignVCenter
                }
            }
            model: result_data.image_names
            spacing: 5
            delegate: Text {
                text: modelData
                verticalAlignment: Text.AlignVCenter
                height: contentHeight * 1.2

                anchors {
                    left: parent ? parent.left : undefined
                    right: parent ? parent.right : undefined
                    margins: 10
                }
                MouseArea {
                    anchors.fill: parent
                    onClicked: imageList.currentIndex = index
                }
            }

            ScrollBar.vertical: ScrollBar { }
        }
    }

    GridLayout {
        id: images
        anchors {
            left: imageListContainer.right
            top: parent.top
            right: parent.right
            bottom: parent.bottom
            margins: 10
        }
        columnSpacing: 10
        rowSpacing: 10

        columns: Math.round(Math.sqrt(children.length))

        ImageViewer {
            id: reference
            imageSource: root.image_path(imageList.currentImage, "reference")
            suffix: "reference"
            imageScale: scaleSlider.value

            Layout.fillWidth: true
            Layout.fillHeight: true

            Rectangle {
                anchors {
                    left: parent.left
                    right: parent.right
                    bottom: parent.bottom
                }
                height: 25
                color: Qt.rgba(0,0,0,0.3)

                Slider {
                    id: scaleSlider

                    anchors {
                        left: parent.left
                        right: parent.right
                        verticalCenter: parent.verticalCenter
                    }
                    handle {
                        height: scaleSlider.height * 0.3
                        width: scaleSlider.height * 0.3
                    }

                    from: Math.min(reference.width/reference.sourceSize.width,
                                   reference.height/reference.sourceSize.height)
                    to: 1.
                    value: 1.
                }
            }
        }

        Repeater {
            model: result_data.denoisers

            ImageViewer {
                imageSource: root.image_path(imageList.currentImage, modelData.name)
                suffix: modelData.name
                interactive: false
                contentX: reference.contentX
                contentY: reference.contentY
                imageScale: scaleSlider.value

                Layout.fillWidth: true
                Layout.fillHeight: true

                Binding {
                    target: modelData
                    property: "image"
                    value: imageList.currentImage
                }
                Component.onCompleted: {
                    for (var key in modelData.metrics)
                        console.log(key + ": " + modelData.metrics[key])
                }
            }
        }
    }
}
