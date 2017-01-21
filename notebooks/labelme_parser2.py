from xml.dom import minidom


def parse_labelme2_xml(xml_string):
    xmldoc = minidom.parseString(xml_string)
    polygons_data = []
    def get_value(node,sub_node_name):
        sub_node = node.getElementsByTagName(sub_node_name)[0].firstChild
        if sub_node:
            return sub_node.nodeValue
        return None

    for object in xmldoc.getElementsByTagName('object'):
        polygon_label = get_value(object, "name")
        polygon_id = get_value(object, "id")
        polygon_status = get_value(object, "attributes")
        points = []
        for point_node in object.getElementsByTagName("pt"):
            x = get_value(point_node, "x")
            y = get_value(point_node, "y")
            points.append([int(x), int(y)])

        polygons_data.append(
            {"parkingSlotLocation": {"points": points}, "name": polygon_label,
                                     "id": polygon_id,
                                     "status": polygon_status})

    return polygons_data
