from ctree.frontend import get_ast
from ctree.nodes import *
from ctree.c.nodes import *
from ctree.c.types import *
from ctree.visual.dot_manager import DotManager

from ctree.dotgen import to_dot
from svm.svm import SVM
def main():
    SVM = SVM()
    pytree = get_ast(SVM.pytrain)
    tree = CFile("train",pytree)
    DotManager.dot_ast_to_browser(tree,'xx.png')


if __name__ == '__main__':
    main()
