import QtQuick 2.15
import QtGraphicalEffects 1.0

Rectangle {
    id: container

    property alias imageSource: image.source
    property string suffix: ""
    property alias sourceSize: image.sourceSize
    property double imageScale: 1.

    property alias contentX: imageContainer.contentX
    property alias contentY: imageContainer.contentY
    property alias interactive: imageContainer.interactive

    border.color: "black"
    color:  "black"

    width: 300
    height: 300

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
}
