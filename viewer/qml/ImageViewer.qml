import QtQuick 2.15
import QtQuick.Layouts 1.2
import QtGraphicalEffects 1.0

Rectangle {
    id: container

    property alias imageSource: image.source
    property string suffix: ""
    property alias sourceSize: image.sourceSize
    property double imageScale: 1.

    property variant metrics: null

    property alias contentX: imageContainer.contentX
    property alias contentY: imageContainer.contentY
    property alias interactive: imageContainer.interactive

    border.color: "black"
    color:  "black"

    width: 300
    height: 300

    onMetricsChanged: {
        metricsModel.clear()
        if (container.metrics != null)  {
            for (var metric in container.metrics) {
                metricsModel.append({metric: metric, value: container.metrics[metric]})
            }
        }
    }

    Flickable {
        id: imageContainer

        anchors {
            fill: parent
            margins: 1
        }

        contentWidth: image.width
        contentHeight: image.height

        clip: true
        boundsBehavior: Flickable.StopAtBounds
        boundsMovement: Flickable.StopAtBounds
        Image {
            id: image

            x: 0
            y: 0
            asynchronous: true
            fillMode: Image.PreserveAspectFit
            smooth: false
            width: sourceSize.width * container.imageScale
            height: sourceSize.height * container.imageScale
        }
    }

    Text {
        anchors {
            top: parent.top
            left: parent.left
            margins: 3
        }
        text: container.suffix
        width: contentWidth
        height: contentHeight
        color: "white"
        layer.enabled: true
        layer.effect: DropShadow {
            verticalOffset: 2
            color: "black"
            radius: 1
            samples: 3
        }
    }

    Rectangle {
        color: Qt.rgba(0,0,0,0.3)

        anchors {
            left: parent.left
            right: parent.right
            bottom: parent.bottom
            margins: 1
        }

        height: childrenRect.height + 10
        visible: container.metrics

        GridLayout {
            columns: 2
            anchors {
                left: parent.left
                right: parent.right
                bottom: parent.bottom
                margins: 5
            }
            height: childrenRect.height

            Repeater {
                model: ListModel { id: metricsModel }
                Text {
                    text: model.metric + ": " + model.value

                    height: paintedHeight
                    Layout.fillWidth: true

                    color: "white"
                    layer.enabled: true
                    layer.effect: DropShadow {
                        verticalOffset: 2
                        color: "black"
                        radius: 1
                        samples: 3
                    }
                }
            }
        }
    }

}
