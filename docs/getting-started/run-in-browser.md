# Run a portable lab in your browser

The course's Python examples use small synthetic inputs. You can paste a complete lab script into a Colab notebook and run it on a CPU. You should use the lesson's linked source file so the code and explanation stay together.

## Google Colab can run the portable examples.

1. You should open [Google Colab](https://colab.research.google.com/) and create a notebook.
2. You should copy one complete Python lab from the repository into a code cell.
3. You should select **Runtime → Run all**.
4. You should compare the output with the lesson's expected behavior and the saved evidence.

The examples use ordinary Python and NumPy unless their setup names another dependency. A native C or CUDA source file is not a Python cell. Its lesson gives a separate compile command and hardware requirement.

## The integration notebook has direct run buttons.

The capstone notebook has Colab and Binder buttons in the website toolbar. These buttons point to the public GitHub source. Binder startup depends on a third-party service and can take several minutes. A successful local notebook run does not guarantee service availability.

## You should preserve your work.

A browser session can end without saving local files. You should save your notebook or download it before leaving. The teaching scripts do not require private data, credentials, or access to the university's Canvas sites.
