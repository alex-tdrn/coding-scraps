#include "DepthBuffering.h"

#include <iostream>

int main()
{
	DepthBuffering app;

	try
	{
		app.run();
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return 1;
	}
	return 0;
}
