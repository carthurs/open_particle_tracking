from __future__ import print_function
from builtins import str
from builtins import range
from xml.etree.ElementTree import Element, SubElement, Comment, tostring
from xml.dom import minidom
import re
import numpy as np

def prettyPrintXML(elem):
    """Return a pretty-printed XML string for the Element.
       Taken from https://pymotw.com/2/xml/etree/ElementTree/create.html
    """
    rough_string = tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def convertListToString(list):
    """Convert list to string and strip out everything not alphanumeric or underscore
       Adding a space after the \w keeps a white space when replacing with ''
    """
    return re.sub(r'[^\w ]', '', str(list))


def example():
    """Example taken from https://pymotw.com/2/xml/etree/ElementTree/create.html
    """
    top = Element('top')

    comment = Comment('Generated for PyMOTW')
    top.append(comment)

    child = SubElement(top, 'child')
    child.text = 'This child contains text.'

    child_with_tail = SubElement(top, 'child_with_tail')
    child_with_tail.text = 'This child has regular text.'
    child_with_tail.tail = 'And "tail" text.'

    child_with_entity_ref = SubElement(top, 'child_with_entity_ref')
    child_with_entity_ref.text = 'This & that'

    print(prettyPrintXML(top))

def generateParticleXDMF():

    # example()

    # root
    root = Element('Xdmf',{'Version':'2.0','xmlns:xi':'http://www.w3.org/2001/XInclude'})

    # domain
    domain = SubElement(root, 'Domain')

    # time series
    timeSeries = SubElement(domain, 'Grid', {'Name':'TimeSeries', 'GridType':'Collection', 'CollectionType':'Temporal'})
    time = SubElement(timeSeries, 'Time', {'TimeType':'List'})

    # stepIterator = xrange(5250, 7750, 2)
    stepIterator = range(5250, 5260, 2)

    data = SubElement(time, 'DataItem', {'Format':'XML', 'Dimensions':str(len(stepIterator))})
    data.text = convertListToString(list(stepIterator))

    for step in stepIterator:
        grid = SubElement(timeSeries, 'Grid', {'Name': 'grid_'+str(step), 'GridType':'Uniform'})
        topology = SubElement(grid, 'Topology', {'NumberOfElements':'1057623', 'TopologyType':'Polyvertex'})



    # for time in xrange(0,10):
    #     print time


    print(prettyPrintXML(root))

if __name__ == '__main__':

    generateParticleXDMF()
