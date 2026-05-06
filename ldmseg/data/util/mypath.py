

import os


class MyPath(object):
    @staticmethod
    def db_root_dir(database='', prefix='/efs/datasets/'):

        db_names = {'coco', 'cityscapes'}
        assert (database in db_names), 'Database {} not available.'.format(database)

        return os.path.join(prefix, database)
