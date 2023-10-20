import time
import tensorflow as tf
from src.functions import *
import imageio

class op(object):

    def __init__(self, args, sess):
        self.sess = sess

        ## Train
        self.gpu_number = args.gpu_number
        self.project_name = args.project

        ## Train images
        self.content_dataset = args.content_dataset ## test2015
        self.content_data_size = args.content_data_size
        self.style_image = args.style_image

        ## Train Iteration
        self.niter = args.niter
        self.niter_snapshot = args.nsnapshot
        self.max_to_keep = args.max_to_keep

        ## Train Parameter
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.momentum2 = args.momentum2

        self.content_weights = args.content_loss_weights
        self.style_weights = args.style_loss_weights
        self.tv_weight = args.tv_loss_weight

        ## Result Dir & File
        self.project_dir = '{0}/'.format(self.project_name)
        make_project_dir(self.project_dir)
        self.ckpt_dir = os.path.join(self.project_dir, 'models')

        ## Test
        self.test_dataset = args.test_dataset
        self.style_control = args.style_control_weights

        ## build model
        self.build_model()


    def train(self,Train_flag):
        data = data_loader(self.content_dataset)
        print('Shuffle ....')
        random_order = np.random.permutation(len(data))
        data = [data[i] for i in random_order[:10000*self.batch_size]]
        print('Shuffle Done')

        start_time = time.time()
        count = 0

        try:
            self.load()
            print('Weight Load !!')
        except:
            self.sess.run(tf.compat.v1.global_variables_initializer())

        # for epoch in range(self.niter):
        for epoch in range(5):
            batch_idxs = len(data) // self.batch_size

            # for idx in range(0, batch_idxs):
            for idx in range(5):
                count += 1

                batch_files = data[idx * self.batch_size: (idx + 1) * self.batch_size]
                batch_label = [(get_image(batch_file, self.content_data_size)) for batch_file in batch_files]

                feeds = {self.content_input: batch_label}

                _, loss_all, loss_c, loss_s, loss_tv = self.sess.run(self.optimize, feed_dict=feeds)
                train_time = time.time() - start_time
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, loss: %.4f, loss_c: %.4f, loss_s: %.4f, loss_tv: %.4f"
                      % (epoch, idx, batch_idxs, train_time, loss_all, loss_c, loss_s, loss_tv))

                ## Test during Training
                print("count: ", count)
                print(self.niter_snapshot)
                if count % self.niter_snapshot == (self.niter_snapshot-1):
                    print("SAVING!!!!!!!!!!!!!!!")
                    self.count = count
                    self.save()
                    self.test(Train_flag)


    def test(self, Train_flag=True):
        for fn in os.listdir(self.test_dataset):

            ## Read RGB Image
            im_input = get_image(self.test_dataset + '/' + fn)
            im_input_4d = im_input[np.newaxis, ...]
            im_b, im_h, im_w, im_c = np.shape(im_input_4d)

            ## Run Model
            img = tf.compat.v1.placeholder(tf.float32, [im_b, im_h, im_w, im_c], name='img')

            self.test_recon = self.mst_net(img, style_control=self.style_control, reuse=True)
            self.load()

            im_output = self.sess.run(self.test_recon, feed_dict={img : im_input_4d})
            im_output = inverse_image(im_output[0])
            # im_output = np.random.random_sample(im_output.shape) * 255
            im_output = im_output.astype(np.uint8)
            print("the order of styles")
            print(self.style_control)
            style_idx = ['{0}_{1}'.format(i, x) for i, x in enumerate(self.style_control) if not x == 0]

            ## Image Show & Save
            style_name = os.path.split(self.style_image)[-1].split('.')[0]
            if Train_flag:
                train_output_dir = os.path.join(self.project_dir, 'train_result', style_name)
                if not os.path.exists(train_output_dir):
                    os.makedirs(train_output_dir)
                filename = fn[:-4] + '_' + str(style_idx) + '_' + str(int(self.count)) + '_output.jpg'
                print("trying to read image, filename: ", os.path.join(train_output_dir, filename), im_output)
                imageio.imwrite(os.path.join(train_output_dir, filename), im_output)
            else:
                test_output_dir = os.path.join(self.project_dir, 'test_result')
                filename = fn[:-4] + '_' + str(style_idx) + '_output.jpg'
                print(type(im_output))
                print(im_output.shape)
                print("trying to read image, filename: ", os.path.join('.', test_output_dir, filename), im_output)
                imageio.imwrite(os.path.join('.', test_output_dir, filename), im_output)

            print(filename)


    def save(self):
        print(" [*] Saving checkpoints...")
        style_name = os.path.basename(self.style_image)[:-4]
        self.model_name = "{0}_{1}.model".format(self.project_name, style_name)

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        self.saver.save(self.sess, os.path.join(self.ckpt_dir, self.model_name), global_step=self.count)


    def load(self):
        print(" [*] Reading checkpoints...", self.ckpt_dir)
        print(self.project_dir)
        ckpt = tf.compat.v1.train.get_checkpoint_state(self.ckpt_dir)
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print("saved model name : ", ckpt_name)
        print(os.path.join(self.ckpt_dir, ckpt_name))
        self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))